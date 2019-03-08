#include "vectorized.hpp"


void Vectorized::map_not_match_void(bool* R out, void** R a,
		void* R b, int* R sel, int num) {
	map(sel, num, [&] (auto i) { out[i] = a[i] != b; });
}

int Vectorized::select_match(int* R osel, bool* R b, int* R sel,
		int num) {
	return select(osel, sel, num, [&] (auto i) { return b[i]; });
}

int Vectorized::select_not_match(int* R osel, bool* R b, int* R sel,
		int num) {
	return select(osel, sel, num, [&] (auto i) { return !b[i]; });
}

int Vectorized::select_match_bit(int* R osel, uint8_t* R a, int num) {
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

void Vectorized::map_hash(uint32_t* R out, int32_t* R a, int* R sel,
		int num) {
	map(sel, num, [&] (auto i) { out[i] = hash32((uint32_t)(a[i])); });
}

void Vectorized::glob_sum(int64_t* R out, int32_t* R a, int* R sel,
		int num) {

	int64_t p=0;
	map(sel, num, [&] (auto i) { p+= a[i]; });
	*out = *out + p;
}
