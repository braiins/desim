//simulate the scheduling of two processes on one CPU
#![feature(generators, generator_trait)]
extern crate rand;
extern crate desim;

use rand::{Rng as RngT, XorShiftRng as Rng};

use desim::{Simulation, Effect, Event, Context};

enum Message {
    Message1,
    Message2
}

fn main(){
    let ctx = Context::<Message>::new();
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
        loop{
            // wait for the CPU
            yield Effect::Request(cpu);
            // do some job for a random amount of time units between 0 and 10
            yield Effect::TimeOut((rng.next_u32() % 10) as f64);
            // release the CPU
            yield Effect::Release(cpu);
        }
    }));
    // let p1 to start immediately...
    s.schedule_event(Event{time: 0.0, process: p1});
    // ...and p2 after 17 time units
    s.schedule_event(Event{time: 17.0, process: p2});
}
