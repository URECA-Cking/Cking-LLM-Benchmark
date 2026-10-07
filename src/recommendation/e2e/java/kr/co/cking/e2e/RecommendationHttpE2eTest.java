package kr.co.cking.e2e;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.*;
import java.nio.charset.StandardCharsets;
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
class RecommendationHttpE2eTest {
    @DynamicPropertySource
    static void isolatedProperties(DynamicPropertyRegistry registry) {
        String url = required("CKING_E2E_JDBC");
        if (!url.matches("jdbc:mysql://127\\.0\\.0\\.1:[0-9]+/cking_e2e_[0-9a-f]+\\?.*")) {
            throw new IllegalArgumentException("E2E owned database required");
        }
        Map<String, String> properties = Map.ofEntries(
            Map.entry("spring.profiles.active", "test"),
            Map.entry("spring.datasource.url", url),
            Map.entry("spring.datasource.username", "root"),
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
    private final ObjectMapper json = new ObjectMapper();
    private final Map<Long, Long> fixtureCreators = new HashMap<>();
    private long fanId;
    private long adminId;

    @Test
    void realHttpAndDatabaseContract() throws Exception {
        // Flyway 시드 외 업무 행이 있다면 fixture 생성 전에 거부한다.
        assertThat(jdbc.queryForObject("select count(*) from member", Long.class)).isZero();
        assertThat(jdbc.queryForObject("select count(*) from creator", Long.class)).isZero();
        String suffix = UUID.randomUUID().toString().substring(0, 8);
        Member fan = member("e2e-fan-" + suffix, MemberRole.USER);
        Member admin = member("e2e-admin-" + suffix, MemberRole.ADMIN);
        fanId = fan.getMemberId();
        adminId = admin.getMemberId();
        // fixture의 논리 ID와 실제 DB ID를 구분한다. 작은 ID도 예약하지 않는다.
        for (long logical : List.of(1L, 2L, 3L, 4L, 5L, 6L, 10L, 101L, 102L, 103L, 104L)) {
            Member owner = logical == 1 ? fan : member("e2e-" + logical + "-" + suffix, MemberRole.USER);
            Creator creator = creators.saveAndFlush(new Creator(owner.getMemberId(), "e2e-" + logical + "-" + suffix));
            fixtureCreators.put(logical, creator.getCreatorId());
            CreatorSpaceTemplate template = new CreatorSpaceTemplate(owner.getMemberId(),
                    "E2E deterministic synthetic creator introduction " + logical, "profile", "banner", "creator-{creatorId}");
            spaces.saveAndFlush(CreatorSpace.fromTemplate(creator.getCreatorId(), template, "e2e-" + creator.getCreatorId()));
        }
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("baseUrl", "http://127.0.0.1:" + port);
        config.put("apiKey", required("CKING_E2E_API_KEY"));
        config.put("userToken", issuer.issue(fanId, MemberRole.USER).accessToken());
        config.put("adminToken", issuer.issue(adminId, MemberRole.ADMIN).accessToken());
        config.put("creatorIds", fixtureCreators);
        config.put("taxonomyHash", jdbc.queryForObject(
                "select taxonomy_hash from interest_taxonomy where taxonomy_version = 'v0.2'", String.class));
        ProcessBuilder builder = new ProcessBuilder(required("CKING_E2E_PYTHON"), "-m", "src.recommendation.e2e.scenarios");
        builder.directory(new File(required("CKING_E2E_LLM_ROOT")));
        // 원문·토큰은 Gradle 로그로 전달하지 않는다. 결과는 비밀값 없는 report.json에만 남긴다.
        builder.redirectError(ProcessBuilder.Redirect.DISCARD);
        Process child = builder.start();
        ScheduledExecutorService timer = Executors.newSingleThreadScheduledExecutor();
        timer.schedule(child::destroyForcibly, 5, TimeUnit.MINUTES);
        try (var input = new BufferedReader(new InputStreamReader(child.getInputStream(), StandardCharsets.UTF_8));
             var output = new BufferedWriter(new OutputStreamWriter(child.getOutputStream(), StandardCharsets.UTF_8))) {
            send(output, config);
            String line;
            while ((line = input.readLine()) != null) {
                JsonNode request = json.readTree(line);
                send(output, control(request));
            }
            assertThat(child.waitFor(10, TimeUnit.SECONDS)).isTrue();
            assertThat(child.exitValue()).as("See LLM E2E report.json (no raw HTTP bodies logged)").isZero();
            assertThat(jdbc.queryForObject("select count(*) from member", Long.class)).isEqualTo(12L);
            assertThat(jdbc.queryForObject("select count(*) from creator", Long.class)).isEqualTo(11L);
        } finally {
            child.destroyForcibly();
            timer.shutdownNow();
        }
    }

    private Member member(String name, MemberRole role) {
        return members.saveAndFlush(new Member(name, null, null, role));
    }

    private void send(BufferedWriter output, Object value) throws Exception {
        output.write(json.writeValueAsString(value));
        output.newLine();
        output.flush();
    }

    private Object control(JsonNode request) {
        return switch (request.path("op").asText()) {
            case "snapshot" -> Map.of(
                "similar", jdbc.queryForList("""
                    select g.creator_id, g.application_sequence, g.input_hash, g.generation_id,
                      (select count(*) from creator_similarity_candidate c where c.generation_id=g.generation_id) candidate_count
                    from creator_similarity_state s join creator_similarity_generation g
                    on g.generation_id=s.current_generation_id order by g.creator_id
                    """),
                "interests", jdbc.queryForList("""
                    select g.interest_code, g.application_sequence, g.input_hash, g.generation_id,
                      (select count(*) from interest_recommendation_candidate c where c.generation_id=g.generation_id) candidate_count
                    from interest_recommendation_state s join interest_recommendation_generation g
                    on g.generation_id=s.current_generation_id where g.taxonomy_version='v0.2' order by g.interest_code
                    """),
                "similarCandidates", jdbc.queryForList("""
                    select s.creator_id, c.similar_creator_id, c.rank_no, c.score
                    from creator_similarity_state s join creator_similarity_candidate c
                    on c.generation_id=s.current_generation_id order by s.creator_id, c.rank_no
                    """),
                "interestCandidates", jdbc.queryForList("""
                    select s.interest_code, c.creator_id, c.rank_no, c.score
                    from interest_recommendation_state s join interest_recommendation_candidate c
                    on c.generation_id=s.current_generation_id where s.taxonomy_version='v0.2'
                    order by s.interest_code, c.rank_no
                    """),
                "generationCounts", List.of(
                    jdbc.queryForObject("select count(*) from creator_similarity_generation", Long.class),
                    jdbc.queryForObject("select count(*) from interest_recommendation_generation", Long.class)));
            case "intro" -> {
                jdbc.update("update creator_space set intro_text=? where creator_id=?",
                    request.path("changed").asBoolean() ? "Changed synthetic E2E introduction" :
                    "E2E deterministic synthetic creator introduction 1", fixtureCreators.get(1L));
                yield Map.of("ok", true);
            }
            case "revokeAdmin" -> {
                jdbc.update("update member set role='USER' where member_id=?", adminId);
                yield Map.of("ok", true);
            }
            case "popularity" -> {
                for (long logical : List.of(3L, 4L, 6L)) {
                    long owner = creators.findById(fixtureCreators.get(logical)).orElseThrow().getMemberId();
                    long target = fixtureCreators.get(logical == 6L ? 102L : 101L);
                    jdbc.update("insert into creator_follow (member_id, creator_id, created_at) values (?, ?, UTC_TIMESTAMP(6))",
                            owner, target);
                }
                yield Map.of("ok", true);
            }
            case "fixtureCandidates" -> {
                // 공유 정책 fixture는 API의 연속 Top-100 범위를 넘어서는 희소 rank도 포함한다.
                // API로 활성화한 빈 세대에 fixture 후보를 준비하고 조회 HTTP·SQL·정책을 검증한다.
                // 실제 모델 생성 후보의 적재/멱등/복귀는 앞선 독립 시나리오에서 HTTP로만 검증한다.
                JsonNode data = request.path("input");
                for (JsonNode source : data.path("interestSources")) {
                    String code = source.path("interestCode").asText();
                    Long generation = jdbc.queryForObject("select current_generation_id from interest_recommendation_state "
                        + "where taxonomy_version='v0.2' and interest_code=?", Long.class, code);
                    for (JsonNode candidate : source.path("activeGeneration").path("candidates")) {
                        jdbc.update("insert into interest_recommendation_candidate (generation_id, creator_id, score, rank_no) "
                            + "values (?, ?, 0.5, ?)", generation, candidate.path("creatorId").asLong(), candidate.path("rank").asInt());
                    }
                }
                for (JsonNode source : data.path("followSources")) {
                    long seed = source.path("seedCreatorId").asLong();
                    if (!fixtureCreators.containsValue(seed)) throw new IllegalArgumentException("fixture seed required");
                    Long generation = jdbc.queryForObject("select current_generation_id from creator_similarity_state "
                        + "where creator_id=?", Long.class, seed);
                    for (JsonNode candidate : source.path("activeGeneration").path("candidates")) {
                        jdbc.update("insert into creator_similarity_candidate (generation_id, similar_creator_id, score, rank_no) "
                            + "values (?, ?, 0.5, ?)", generation, candidate.path("creatorId").asLong(), candidate.path("rank").asInt());
                    }
                }
                yield Map.of("ok", true);
            }
            case "personalization" -> {
                // 삭제는 이 실행에서 생성한 회원 관계에만 한정한다.
                jdbc.update("delete from member_interest where member_id=?", fanId);
                jdbc.update("delete from creator_follow where member_id=?", fanId);
                for (JsonNode code : request.path("interests")) {
                    jdbc.update("insert into member_interest values (?, 'v0.2', ?, UTC_TIMESTAMP(6))", fanId, code.asText());
                }
                for (JsonNode id : request.path("followed")) {
                    jdbc.update("insert into creator_follow (member_id, creator_id, created_at) values (?, ?, UTC_TIMESTAMP(6))",
                            fanId, id.asLong());
                }
                // Space 없는 후보를 재현하도록 이 실행 fixture의 Space만 제거/복원한다.
                for (var entry : fixtureCreators.entrySet()) {
                    long id = entry.getValue();
                    boolean exists = false;
                    for (JsonNode space : request.path("spaces")) {
                        if (space.path("creatorId").asLong() == id && space.path("hasCreatorSpace").asBoolean()) exists = true;
                    }
                    if (!exists) jdbc.update("delete from creator_space where creator_id=?", id);
                    else if (spaces.findByCreatorId(id).isEmpty()) {
                        long owner = creators.findById(id).orElseThrow().getMemberId();
                        spaces.saveAndFlush(CreatorSpace.fromTemplate(id,
                            new CreatorSpaceTemplate(owner, "synthetic", "profile", "banner", "creator-{creatorId}"), "e2e-" + id));
                    }
                }
                yield Map.of("ok", true);
            }
            default -> throw new IllegalArgumentException("Unknown E2E control operation");
        };
    }
}
