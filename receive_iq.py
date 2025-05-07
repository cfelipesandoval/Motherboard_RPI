from iq_receiver import IQReceiver

def main():
  receiver = IQReceiver('/dev/ttyACM0', channel_mode=2)

  batch_size = 128
  receiver.collect_samples(batch_size)

  try:
    samples = receiver.read_batch(batch_size)
    for idx, row in enumerate(samples):
      if receiver.channel_mode == 2:
        (iA, qA, tA), (iB, qB, tB) = row
        print(f"{idx}: ChA [I={iA}, Q={qA}, t={tA}] | ChB [I={iB}, Q={qB}, t={tB}]")
      else:
        iA, qA, tA = row
        print(f"{idx}: ChA [I={iA}, Q={qA}, t={tA}]")

  finally:
    receiver.close()


if __name__ == '__main__':
  main()