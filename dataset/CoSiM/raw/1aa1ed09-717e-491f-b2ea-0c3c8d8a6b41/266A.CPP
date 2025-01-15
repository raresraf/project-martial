#include <iostream>
using namespace std;

int main() {
  int n, result, R, G, B;
  R = G = B = 0;
  cin >> n;
  string input;
  cin >> input;

  for (int i = 0; i < input.length() - 1; i++) {
    if (input[i] == input[i + 1])
      if (input[i] == 'R')
        R++;
      else if (input[i] == 'G')
        G++;
      else if (input[i] == 'B')
        B++;
  }

  result = R + G + B;
  cout << result << endl;

  return 0;
}