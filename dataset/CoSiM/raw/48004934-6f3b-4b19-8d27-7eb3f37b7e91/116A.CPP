#include <iostream>

using namespace std;

int main() {
  int number;
  cin >> number;
  int in[number], out[number], i, total = 0, minCap = 0;

  for (i = 0; i < number; i++) {
    cin >> out[i] >> in[i];
  }

  for (i = 0; i < number; i++) {
    total = total + in[i] - out[i];
    if (total > minCap)
      minCap = total;
  }

  cout << minCap;
  return 0;
}
