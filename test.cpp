#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("oom1", "[alloc][oom]" ) {

	SECTION("64MB") {
		REQUIRE(true);
	}

}