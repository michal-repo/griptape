---
search:
  boost: 2 
---

## Overview

You can persist and load memory by using Conversation Memory Drivers. You can build drivers for your own data stores by extending [BaseConversationMemoryDriver](../../reference/griptape/drivers/memory/conversation/base_conversation_memory_driver.md).

## Conversation Memory Drivers

### Local

The [LocalConversationMemoryDriver](../../reference/griptape/drivers/memory/conversation/local_conversation_memory_driver.md) allows you to persist Conversation Memory in a local JSON file.

```python
from griptape.structures import Agent
from griptape.drivers import LocalConversationMemoryDriver
from griptape.memory.structure import ConversationMemory

local_driver = LocalConversationMemoryDriver(file_path="memory.json")
agent = Agent(conversation_memory=ConversationMemory(driver=local_driver))

agent.run("Surfing is my favorite sport.")
agent.run("What is my favorite sport?")
```

### Amazon DynamoDb

!!! info
    This driver requires the `drivers-memory-conversation-amazon-dynamodb` [extra](../index.md#extras).

The [AmazonDynamoDbConversationMemoryDriver](../../reference/griptape/drivers/memory/conversation/amazon_dynamodb_conversation_memory_driver.md) allows you to persist Conversation Memory in [Amazon DynamoDb](https://aws.amazon.com/dynamodb/).

```python
import os
import uuid
from griptape.drivers import AmazonDynamoDbConversationMemoryDriver
from griptape.memory.structure import ConversationMemory
from griptape.structures import Agent

conversation_id = uuid.uuid4().hex
dynamodb_driver = AmazonDynamoDbConversationMemoryDriver(
    table_name=os.environ["DYNAMODB_TABLE_NAME"],
    partition_key="id",
    value_attribute_key="memory",
    partition_key_value=conversation_id,
)

agent = Agent(conversation_memory=ConversationMemory(driver=dynamodb_driver))

agent.run("My name is Jeff.")
agent.run("What is my name?")
```
Optional parameters `sort_key` and `sort_key_value` can be supplied for tables with a composite primary key.


### Redis

!!! info
    This driver requires the `drivers-memory-conversation-redis` [extra](../index.md#extras).

The [RedisConversationMemoryDriver](../../reference/griptape/drivers/memory/conversation/redis_conversation_memory_driver.md) allows you to persist Conversation Memory in [Redis](https://redis.io/).

```python
import os
import uuid
from griptape.drivers import RedisConversationMemoryDriver
from griptape.memory.structure import ConversationMemory
from griptape.structures import Agent

conversation_id = uuid.uuid4().hex
redis_conversation_driver = RedisConversationMemoryDriver(
    host=os.environ["REDIS_HOST"],
    port=os.environ["REDIS_PORT"],
    password=os.environ["REDIS_PASSWORD"],
    index=os.environ["REDIS_INDEX"],
    conversation_id = conversation_id
)

agent = Agent(conversation_memory=ConversationMemory(driver=redis_conversation_driver))

agent.run("My name is Jeff.")
agent.run("What is my name?")
```
