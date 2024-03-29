/**
 * Copyright 2007 The Apache Software Foundation
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hama.bsp;

import java.io.Closeable;
import java.io.IOException;

import java.net.InetSocketAddress;

import org.apache.zookeeper.KeeperException;

/**
 * 
 */
public interface DefaultBSPPeer extends Closeable, BSPConstants {

  /**
   * Send a data with a tag to another BSPSlave corresponding to hostname.
   * Messages sent by this method are not guaranteed to be received in a sent
   * order.
   * 
   * @param hostname
   * @param msg
   * @throws IOException
   */
  public void send(InetSocketAddress hostname, BSPMessage msg)
      throws IOException;

  /**
   * @return the current message
   * @throws IOException
   */
  public BSPMessage getCurrentMessage() throws IOException;

  /**
   * Synchronize all of the data in the local queue to other BSP Peers.
   * 
   * @throws InterruptedException
   * @throws KeeperException
   */
  public void sync() throws IOException, KeeperException, InterruptedException;
}
