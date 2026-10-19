package kr.co.cking.e2e;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import kr.co.cking.auth.application.port.AccessTokenIssuer;
import kr.co.cking.creator.domain.*;
import kr.co.cking.creator.repository.*;
import kr.co.cking.member.domain.*;
import kr.co.cking.member.repository.MemberRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

/** HTTP 인증은 실제 필터를 통과한다. 제어 채널은 자식 Python의 stdin/stdout이며 HTTP API가 아니다. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class ContentTagHttpE2eTest {
    @DynamicPropertySource
    static void isolatedProperties(DynamicPropertyRegistry registry) {
        String url = required("CKING_E2E_JDBC");
        if (!url.matches("jdbc:mysql://127\\.0\\.0\\.1:[0-9]+/cking_e2e_[0-9a-f]+\\?.*")) {
            throw new IllegalArgumentException("E2E owned database required");
        }
        Map<String, String> properties = Map.ofEntries(
            Map.entry("spring.profiles.active", "test"),
            Map.entry("spring.datasource.url", url),
            Map.entry("spring.datasource.username", required("CKING_E2E_DB_USERNAME")),
            Map.entry("spring.datasource.password", required("CKING_E2E_DB_PASSWORD")),
            Map.entry("spring.data.redis.host", "127.0.0.1"),
            Map.entry("spring.data.redis.port", required("CKING_E2E_REDIS_PORT")),
            Map.entry("management.server.port", "0"),
            Map.entry("server.address", "127.0.0.1"),
            Map.entry("management.server.address", "127.0.0.1"),
            Map.entry("cking.storage.type", "memory"),
            Map.entry("cking.auth.frontend-callback-url", "http://127.0.0.1/callback"),
            Map.entry("cking.auth.refresh-cookie-secure", "false"),
            Map.entry("cking.auth.jwt.secret", required("CKING_E2E_JWT_SECRET")),
            Map.entry("cking.recommendation.api-key-hashes", required("CKING_E2E_KEY_HASH")),
            Map.entry("cking.content-tags.token-hash", required("CKING_E2E_KEY_HASH")),
            Map.entry("cking.scheduling.enabled", "false"),
            Map.entry("cking.verification.youtube-subscription.recovery.enabled", "false"),
            Map.entry("spring.jpa.show-sql", "false"),
            Map.entry("logging.level.root", "WARN")
        );
        properties.forEach((key, value) -> registry.add(key, () -> value));
    }

    private static String required(String name) {
        return Objects.requireNonNull(System.getenv(name), name);
    }

    @LocalServerPort int port;
    @Autowired JdbcTemplate jdbc;
    @Autowired MemberRepository members;
    @Autowired CreatorRepository creators;
    @Autowired CreatorSpaceRepository spaces;
    @Autowired AccessTokenIssuer issuer;
    @Autowired kr.co.cking.creator.application.ContentTagService tags;
    @Autowired kr.co.cking.creator.application.CreatorContentOutbox outbox;
    @Autowired org.springframework.transaction.PlatformTransactionManager transactions;
    private final ObjectMapper json = new ObjectMapper();
    private long creatorId;
    private long ownerId;

    @Test
    void pythonWorkerThroughRealHttpAndMysql() throws Exception {
        assertThat(required("CKING_E2E_DB_USERNAME")).isNotEqualTo("root");
        assertThat(jdbc.queryForList("SHOW GRANTS",String.class)).noneMatch(grant -> grant.contains("SUPER") || grant.contains("ALL PRIVILEGES ON *.*"));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM creator",Long.class)).isZero();
        Member owner = members.saveAndFlush(new Member("content-owner",null,null,MemberRole.USER));
        Member other = members.saveAndFlush(new Member("content-other",null,null,MemberRole.USER));
        List<Long> ids = new ArrayList<>();
        for (Member member : List.of(owner,other)) {
            Creator c = creators.saveAndFlush(new Creator(member.getMemberId(),member.getName()));
            ids.add(c.getCreatorId());
            new org.springframework.transaction.support.TransactionTemplate(transactions).executeWithoutResult(status -> {
                CreatorSpace space=spaces.saveAndFlush(CreatorSpace.fromTemplate(c.getCreatorId(),
                    new CreatorSpaceTemplate(member.getMemberId(),"노래 커버", "profile","banner","creator-{creatorId}"), "content-"+c.getCreatorId()));
                outbox.enqueue(space);
            });
        }
        creatorId=ids.getFirst(); ownerId=owner.getMemberId();
        Map<String,Object> config = Map.of("baseUrl","http://127.0.0.1:"+port,"apiKey",required("CKING_E2E_API_KEY"),
                "userToken",issuer.issue(owner.getMemberId(),MemberRole.USER).accessToken(),
                "otherToken",issuer.issue(other.getMemberId(),MemberRole.USER).accessToken(),"creatorIds",Map.of("1",ids.get(0),"2",ids.get(1)));
        ProcessBuilder builder = new ProcessBuilder(required("CKING_E2E_PYTHON"),"-m","src.recommendation.e2e.content_scenarios");
        builder.directory(new File(required("CKING_E2E_LLM_ROOT")));
        builder.redirectError(new File(required("CKING_E2E_OUTPUT"),"python-private.log"));
        Process child=builder.start();
        var timer=Executors.newSingleThreadScheduledExecutor();timer.schedule(child::destroyForcibly,8,TimeUnit.MINUTES);
        try (var input=new BufferedReader(new InputStreamReader(child.getInputStream(),StandardCharsets.UTF_8));
             var output=new BufferedWriter(new OutputStreamWriter(child.getOutputStream(),StandardCharsets.UTF_8))) {
            send(output,config);
            String line;
            while ((line=input.readLine())!=null) send(output,control(json.readTree(line)));
            assertThat(child.waitFor(15,TimeUnit.SECONDS)).isTrue();
            assertThat(child.exitValue()).as("See protected content E2E report").isZero();
        } finally {child.destroyForcibly();timer.shutdownNow();}
    }
    private void send(BufferedWriter output,Object value) throws Exception {
        output.write(json.writeValueAsString(value));output.newLine();output.flush();
    }
    private Object control(JsonNode p) throws Exception {
        return switch(p.path("op").asText()) {
            case "register" -> {
                JsonNode taxonomy=p.path("taxonomy");
                List<kr.co.cking.interest.domain.InterestSubtopicHash.Row> rows=new ArrayList<>();
                Map<String,String> parents=new LinkedHashMap<>();
                for (JsonNode row:taxonomy.path("subtopics")) {
                    rows.add(new kr.co.cking.interest.domain.InterestSubtopicHash.Row(row.path("code").asText(),row.path("parent").asText(),row.path("name").asText(),row.path("definition").asText()));
                    parents.put(row.path("code").asText(),row.path("parent").asText());
                }
                assertThat(kr.co.cking.interest.domain.InterestSubtopicHash.compute(rows)).isEqualTo(p.path("identity").path("taxonomyHash").asText());
                jdbc.update("INSERT INTO creator_content_taxonomy(taxonomy_version,taxonomy_hash,criteria_hash,config_hash,parent_codes,topic_parents,taxonomy_snapshot,approved,active) VALUES(?,?,?,?,?,?,?,TRUE,TRUE)",
                    taxonomy.path("version").asText(),p.path("identity").path("taxonomyHash").asText(),p.path("identity").path("criteriaHash").asText(),p.path("configHash").asText(),json.writeValueAsString(taxonomy.path("parents")),json.writeValueAsString(parents),json.writeValueAsString(taxonomy));
                yield Map.of("registered",true);
            }
            case "snapshot" -> Map.of("automaticCount",jdbc.queryForObject("SELECT COUNT(*) FROM creator_content_automatic",Long.class),
                "confirmationCount",jdbc.queryForObject("SELECT COUNT(*) FROM creator_content_confirmation",Long.class),
                "jobs",jdbc.queryForList("SELECT creator_id,generation,status,attempt_count FROM creator_content_job ORDER BY creator_id"),
                "similarCount",jdbc.queryForObject("SELECT COUNT(*) FROM creator_similarity_generation",Long.class),
                "interestCount",jdbc.queryForObject("SELECT COUNT(*) FROM interest_recommendation_generation",Long.class));
            case "empty" -> {
                new org.springframework.transaction.support.TransactionTemplate(transactions).executeWithoutResult(status -> {
                    CreatorSpace space=spaces.findByCreatorIdForUpdate(creatorId).orElseThrow();
                    space.updateProfile("",space.getProfileImageUrl(),space.getBannerImageUrl());
                    spaces.saveAndFlush(space);
                    outbox.enqueue(space);
                });
                yield Map.of("legacyEmptyFixture",true);
            }
            case "config" -> {jdbc.update("UPDATE creator_content_taxonomy SET config_hash=?",p.path("configHash").asText());yield Map.of("changed",true);}
            case "expire" -> {jdbc.update("UPDATE creator_content_job SET lease_until=TIMESTAMPADD(SECOND,-1,UTC_TIMESTAMP(6)) WHERE creator_id=?",creatorId);yield Map.of("expired",true);}
            case "ready" -> {jdbc.update("UPDATE creator_content_job SET next_retry_at=UTC_TIMESTAMP(6)");yield Map.of("ready",true);}
            case "race" -> race(p.path("payload"));
            case "disable" -> {jdbc.update("UPDATE creator_content_taxonomy SET approved=FALSE");yield Map.of("disabled",true);}
            default -> throw new IllegalArgumentException("Unknown test control");
        };
    }
    private Object race(JsonNode payload) throws Exception {
        var tx=new org.springframework.transaction.support.TransactionTemplate(transactions);
        ExecutorService executor=Executors.newFixedThreadPool(2);
        var pending=new java.util.concurrent.atomic.AtomicReference<Future<Boolean>>();
        var confirmation=new java.util.concurrent.atomic.AtomicReference<Future<Boolean>>();
        CountDownLatch started=new CountDownLatch(2);
        try {
            tx.executeWithoutResult(status -> {
                CreatorSpace space=spaces.findByCreatorIdForUpdate(creatorId).orElseThrow();
                pending.set(executor.submit(() -> {
                    started.countDown();
                    try { tags.apply(creatorId,json.convertValue(payload,Map.class)); return false; }
                    catch(kr.co.cking.common.exception.BusinessException e) {return e.getErrorCode()==ContentTagErrorCode.CONTENT_INPUT_CONFLICT;}
                }));
                confirmation.set(executor.submit(() -> {
                    started.countDown();
                    try { tags.confirm(ownerId,Map.of("automaticRevision",payload.path("automaticRevision").asText(),"inputHash",payload.path("inputHash").asText(),"parents",List.of("MUSIC"),"topics",List.of("MUSIC_VOCAL"))); return false; }
                    catch(kr.co.cking.common.exception.BusinessException e) {return e.getErrorCode()==ContentTagErrorCode.CONTENT_INPUT_CONFLICT;}
                }));
                try {assertThat(started.await(5,TimeUnit.SECONDS)).isTrue();Thread.sleep(200);assertThat(pending.get().isDone()).isFalse();assertThat(confirmation.get().isDone()).isFalse();}
                catch(InterruptedException e){throw new RuntimeException(e);}
                space.updateProfile("수정된 노래 커버",space.getProfileImageUrl(),space.getBannerImageUrl());
                spaces.saveAndFlush(space);
                outbox.enqueue(space);
            });
            assertThat(pending.get().get(10,TimeUnit.SECONDS)).isTrue();
            assertThat(confirmation.get().get(10,TimeUnit.SECONDS)).isTrue();
            return Map.of("serialized",true);
        } finally {executor.shutdownNow();}
    }
}
