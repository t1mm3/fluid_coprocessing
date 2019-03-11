#include "vectorized.hpp"


void Vectorized::map_not_match_bucket_t(bool* CPU_R out, bucket_t* CPU_R a,
		bucket_t b, int* CPU_R sel, int num) {
	map(sel, num, [&] (auto i) { out[i] = a[i] != b; });
}

int Vectorized::select_match(int* CPU_R osel, bool* CPU_R b, int* CPU_R sel,
		int num) {
	return select(osel, sel, num, [&] (auto i) { return b[i]; });
}

int Vectorized::select_not_match(int* CPU_R osel, bool* CPU_R b, int* CPU_R sel,
		int num) {
	return select(osel, sel, num, [&] (auto i) { return !b[i]; });
}

int Vectorized::select_match_bit(int* CPU_R osel, uint8_t* CPU_R a, int num) {
	int res=0;
	int i=0;
#define A(z) { int i=z; int w=a[i/8]; uint8_t m=1 << (i % 8); if (w & m) { osel[res++] = i;}}

	for (; i+8<num; i+=8) {
		A(i);
		A(i+1);
		A(i+2);
		A(i+3);
		A(i+4);
		A(i+5);
		A(i+6);
		A(i+7);
	}
	for (; i<num; i++) { A(i); }
#undef A
	return res;
}

void Vectorized::map_hash(uint32_t* CPU_R out, int32_t* CPU_R a, int* CPU_R sel,
		int num) {
	map(sel, num, [&] (auto i) { out[i] = hash32((uint32_t)(a[i])); });
}

void Vectorized::glob_sum(int64_t* CPU_R out, int32_t* CPU_R a, int* CPU_R sel,
		int num) {

	int64_t p=0;
	map(sel, num, [&] (auto i) { p+= a[i]; });
	*out = *out + p;
}
