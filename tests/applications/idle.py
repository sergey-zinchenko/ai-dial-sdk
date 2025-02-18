import asyncio
from typing import List

from aidial_sdk.chat_completion import ChatCompletion, Request, Response


class IdleApplication(ChatCompletion):
    """
    Application that waits the given intervals before producing chunks.
    """

    intervals: List[float]
    throw_exception: bool

    def __init__(self, intervals: List[float], throw_exception: bool):
        self.intervals = intervals
        self.throw_exception = throw_exception

    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        # sleep before the first chunk is generated
        await asyncio.sleep(self.intervals[0])

        response.set_response_id("test_id")
        response.set_created(0)

        with response.create_single_choice() as choice:
            choice.append_content("1")
            for idx, interval in enumerate(self.intervals[1:], 2):
                await asyncio.sleep(interval)
                choice.append_content(str(idx))

            if self.throw_exception:
                raise RuntimeError("Something went wrong")
