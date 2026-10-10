import asyncio

from isaac.run_read_only_connector import configure_long_running_timeline, run_gateway_server


def test_gateway_server_uses_thread_owned_event_loop() -> None:
    class Server:
        served = False

        async def serve(self) -> None:
            await asyncio.sleep(0)
            self.served = True

    server = Server()
    run_gateway_server(server)
    assert server.served is True


def test_connector_extends_short_stage_timeline() -> None:
    class Timeline:
        end_time = 0.0
        looping = True

        def set_end_time(self, value: float) -> None:
            self.end_time = value

        def set_looping(self, value: bool) -> None:
            self.looping = value

    timeline = Timeline()
    configure_long_running_timeline(timeline)
    assert timeline.end_time == 1_000_000.0
    assert timeline.looping is True
