#!/usr/bin/python3

import sys, string, os
import matplotlib.pyplot as plt

MAX_PROCESSORS = 8

def getFileNickname(filename):
  # Retrieve portion of file path which only says name
  fileNickname = filename
  nicknameIndex = filename.rfind('/')
  if (nicknameIndex != -1):
    fileNickname = filename[nicknameIndex + 1:]

  # Retrieve portion of name before file extension
  nicknameIndex = fileNickname.find('.')
  if (nicknameIndex != -1):
    fileNickname = fileNickname[:nicknameIndex]

  return fileNickname



def runAlgorithm(algorithm, filename, baselineTime):
  xVals = []
  yVals = []

  numProcessors = 1
  while (numProcessors <= MAX_PROCESSORS):
    xVals.append(numProcessors)
    
    output = os.popen("./" + algorithm + " -p " + str(numProcessors) + " " + filename).read()
    outputLines = output.split("\n")
    outputTime = float(outputLines[0].split(": ")[1][:-2])
    outputColors = int(outputLines[1].split(": ")[1])

    yVals.append(baselineTime / outputTime)

    numProcessors *= 2

  plt.plot(xVals, yVals, label = algorithm, marker = ".")

def runAlgorithms(filename):
  fileNickname = getFileNickname(filename)
  print("Running Algorithms on " + fileNickname)

  # Running of the algorithms
  # Run sequential algorithm separately for baseline numbers
  sequentialOutput = os.popen("./sequential " + filename).read()
  outputLines = sequentialOutput.split("\n")
  baselineTime = float(outputLines[0].split(": ")[1][:-2])
  baselineColors = int(outputLines[1].split(": ")[1])

  # Running rest of the algorithms
  runAlgorithm("jp", filename, baselineTime)
  runAlgorithm("gm", filename, baselineTime)
  runAlgorithm("topology", filename, baselineTime)
  runAlgorithm("data-driven", filename, baselineTime)

  # Setting plot parameters 
  plt.title("Speedup for " + fileNickname)
  plt.xlabel("Number of Processors")
  plt.ylabel("Speedup")
  
  xTicks = [1]
  while(xTicks[-1] < MAX_PROCESSORS):
    xTicks.append(xTicks[-1] * 2)
  plt.xticks(xTicks)
  
  plt.legend()

  plt.savefig("output/" + fileNickname + ".png")
  plt.clf()
  # plt.show()

def main():
  if (len(sys.argv) < 2):
    print("Usage: [file directory]")
    return
  directory = sys.argv[1]
  for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    runAlgorithms(filepath)

if __name__ == '__main__':
  main()
