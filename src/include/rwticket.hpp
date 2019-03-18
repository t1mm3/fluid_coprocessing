#pragma once

// from https://locklessinc.com/articles/locks/
#define atomic_xadd(P, V) __sync_fetch_and_add((P), (V))
#define cmpxchg(P, O, N) __sync_val_compare_and_swap((P), (O), (N))
#define atomic_inc(P) __sync_add_and_fetch((P), 1)
#define atomic_dec(P) __sync_add_and_fetch((P), -1) 
#define atomic_add(P, V) __sync_add_and_fetch((P), (V))
#define atomic_set_bit(P, V) __sync_or_and_fetch((P), 1<<(V))
#define atomic_clear_bit(P, V) __sync_and_and_fetch((P), ~(1<<(V)))

/* Compile read-write barrier */
#define barrier() asm volatile("": : :"memory")

/* Pause instruction to prevent excess processor bus usage */ 
#define cpu_relax() asm volatile("pause\n": : :"memory")

typedef union rwticket rwticket;

union rwticket
{
	unsigned u;
	unsigned short us;
	__extension__ struct
	{
		unsigned char write;
		unsigned char read;
		unsigned char users;
	} s;
};

static void rwticket_wrlock(rwticket *l)
{
	unsigned me = atomic_xadd(&l->u, (1<<16));
	unsigned char val = me >> 16;
	
	while (val != l->s.write) cpu_relax();
}

static void rwticket_wrunlock(rwticket *l)
{
	rwticket t = *l;
	
	barrier();

	t.s.write++;
	t.s.read++;
	
	*(unsigned short *) l = t.us;
}

static int rwticket_wrtrylock(rwticket *l)
{
	unsigned me = l->s.users;
	unsigned char menew = me + 1;
	unsigned read = l->s.read << 8;
	unsigned cmp = (me << 16) + read + me;
	unsigned cmpnew = (menew << 16) + read + me;

	if (cmpxchg(&l->u, cmp, cmpnew) == cmp) return 0;
	
	return EBUSY;
}

static void rwticket_rdlock(rwticket *l)
{
	unsigned me = atomic_xadd(&l->u, (1<<16));
	unsigned char val = me >> 16;
	
	while (val != l->s.read) cpu_relax();
	l->s.read++;
}

static void rwticket_rdunlock(rwticket *l)
{
	atomic_inc(&l->s.write);
}

static int rwticket_rdtrylock(rwticket *l)
{
	unsigned me = l->s.users;
	unsigned write = l->s.write;
	unsigned char menew = me + 1;
	unsigned cmp = (me << 16) + (me << 8) + write;
	unsigned cmpnew = ((unsigned) menew << 16) + (menew << 8) + write;

	if (cmpxchg(&l->u, cmp, cmpnew) == cmp) return 0;
	
	return EBUSY;
}
