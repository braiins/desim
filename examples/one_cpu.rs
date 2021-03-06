//simulate the scheduling of two processes on one CPU
#![feature(generators, generator_trait)]
extern crate desim;
extern crate rand;

use rand::{Rng as RngT, XorShiftRng as Rng};

use desim::{Context, Effect, Simulation, Time, Event};

struct Message();

#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
struct Timestamp(f64);

impl Time for Timestamp {
    type Duration = f64;
    fn add(&self, duration: Self::Duration) -> Self {
        Timestamp(self.0 + duration)
    }
}

fn main() {
    let ctx = Context::<Message, Timestamp>::new();
    let mut s = Simulation::new(&ctx);
    let cpu = s.create_resource(1);
    let p1 = ctx.reserve_pid();
    s.create_process(p1, Box::new(move || {
        for _ in 0..10 {
            // wait for the cpu to be available
            yield Effect::Request(cpu);
            // do some job that requres a fixed amount of 5 time units
            yield Effect::TimeOut(5.0);
            // release the CPU
            yield Effect::Release(cpu);
        }
    }));
    let p2 = ctx.reserve_pid();
    s.create_process(p2, Box::new(move || {
        let mut rng = Rng::new_unseeded();
        loop {
            // wait for the CPU
            yield Effect::Request(cpu);
            // do some job for a random amount of time units between 0 and 10
            yield Effect::TimeOut((rng.next_u32() % 10) as f64);
            // release the CPU
            yield Effect::Release(cpu);
        }
    }));
    // let p1 to start immediately...
    s.schedule_event(Event { delay: 0.0, process: p1 });
    // ...and p2 after 17 time units
    s.schedule_event(Event { delay: 17.0, process: p2 });
}
