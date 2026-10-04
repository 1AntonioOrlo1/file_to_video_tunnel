#!/usr/bin/env python3
"""Standalone local TCP echo server (the 'remote app' at the far end).

Kept as its own process so its lifetime is independent of the test
process: when a timed-out test dies, the echo must not die with it
(orphaned tunnel nodes would then keep dialing a dead port).
"""
import asyncio
import sys


async def main(port):
    def conn(r, w):
        async def go():
            n = 0
            try:
                while True:
                    d = await r.read(65536)
                    if not d:
                        break
                    w.write(d)
                    await w.drain()
                    n += len(d)
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                w.close()
                print(f"echo conn from {w.get_extra_info('peername')} "
                      f"closed after {n} bytes", flush=True)
        return asyncio.ensure_future(go())

    srv = await asyncio.start_server(conn, '127.0.0.1', port)
    print(f"echo listening on 127.0.0.1:{port}", flush=True)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main(int(sys.argv[1])))
    except KeyboardInterrupt:
        pass
