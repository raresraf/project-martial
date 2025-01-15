#include <iostream>

using namespace std;

int main() {
  string summands;
  cin >> summands;
  int inputLength = summands.length();
  int numberOfSummands = (inputLength + 1) / 2;
  for (int i = 0; i < numberOfSummands; i++) {
    for (int j = 2; j < inputLength; j = j + 2) {
      if (summands[j] < summands[j - 2]) {
        int tmp = summands[j];
        summands[j] = summands[j - 2];
        summands[j - 2] = tmp;
      }
    }
  }
  cout << summands << endl;
  return 0;
}