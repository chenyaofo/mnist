import flame
from flame.engine import Engine, Event
from flame.metrics import AverageMetric, AccuracyMetric, TimeMetric
from flame.handlers import TimeEstimater
from flame.utils import create_code_snapshot

from core import Context
from material import ctx

if __name__ == '__main__':
    create_code_snapshot()
    engine = Engine(ctx)

    AccuracyMetric("accuracy", (1, 5), context_map=lambda ctx: (ctx.targets, ctx.outputs)).attach(engine)
    AverageMetric("loss", context_map=lambda ctx: ctx.loss).attach(engine)
    TimeMetric("timer").attach(engine)
    TimeEstimater("eta").attach(engine)


    @engine.epoch_flow_control
    def flow(engine: Engine, ctx: Context):
        engine.run_phase(ctx.train_phase)
        engine.run_phase(ctx.validation_phase)


    @engine.iter_func(ctx.train_phase)
    def train(engine: Engine, ctx: Context):
        datas, targets = ctx.inputs
        ctx.datas, ctx.targets = datas.to(ctx.device), targets.to(ctx.device)
        ctx.outputs = ctx.net(ctx.datas)
        ctx.loss = ctx.criterion(ctx.outputs, targets)
        ctx.optimizer.zero_grad()
        ctx.loss.backward()
        ctx.optimizer.step()

    @engine.iter_func(ctx.train_phase,debug=True)
    def debug_train(engine: Engine, ctx: Context):
        datas, targets = ctx.inputs
        ctx.datas, ctx.targets = datas.to(ctx.device), targets.to(ctx.device)
        ctx.timer.mark("to device")
        ctx.outputs = ctx.net(ctx.datas)
        ctx.timer.mark("forward")
        ctx.loss = ctx.criterion(ctx.outputs, targets)
        ctx.timer.mark("criterion")
        ctx.optimizer.zero_grad()
        ctx.loss.backward()
        ctx.timer.mark("backward")
        ctx.optimizer.step()
        ctx.timer.mark("step")

    @engine.iter_func(ctx.validation_phase)
    def validation(engine: Engine, ctx: Context):
        datas, targets = ctx.inputs
        ctx.datas, ctx.targets = datas.to(ctx.device), targets.to(ctx.device)
        ctx.outputs = ctx.net(ctx.datas)
        ctx.loss = ctx.criterion(ctx.outputs, targets)


    @engine.on(Event.PHASE_STARTED)
    def set_net_training_state(engine: Engine, ctx: Context):
        if ctx.is_in_phase(ctx.train_phase):
            ctx.net.train()
        else:
            ctx.net.eval()


    @engine.on(Event.PHASE_COMPLETED)
    def sche(engine: Engine, ctx: Context):
        if ctx.is_in_phase(ctx.train_phase) and ctx.scheduler is not None:
            ctx.scheduler.step()


    @engine.on(Event.ITER_COMPLETED)
    def iter_log(engine: Engine, ctx: Context):
        flame.logger.info(
            "{}, Epoch={}, Iter={}/{}, Loss={:.4f}, Accuracy@1={:.2f}%, Accuracy@5={:.2f}%, ETA={:.2f}s".format(
                ctx.phase.name.upper(),
                ctx.epoch,
                ctx.iteration,
                ctx.max_iteration,
                ctx.entrypoints["loss"].value,
                ctx.entrypoints["accuracy"][1].rate * 100,
                ctx.entrypoints["accuracy"][5].rate * 100,
                ctx.entrypoints["eta"].value,
            ))


    @engine.on(Event.PHASE_COMPLETED)
    def epoch_log(engine: Engine, ctx: Context):
        flame.logger.info(
            "{} Complete, Epoch={}, Loss={:.4f}, Accuracy@1={:.2f}%, Accuracy@5={:.2f}%, Eplased Time={:.2f}s".format(
                ctx.phase.name.upper(),
                ctx.epoch,
                ctx.entrypoints["loss"].value,
                ctx.entrypoints["accuracy"][1].rate * 100,
                ctx.entrypoints["accuracy"][5].rate * 100,
                ctx.entrypoints["timer"].value,
            ))


    engine.run()
