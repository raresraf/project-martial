#include <iostream>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;

  string input;
  cin >> input;

  for (int k = 0; k < t; k++)
    for (int i = 1; i < n; ++i)
      if (input[i] == 'G' && input[i - 1] == 'B') {
        input[i] = 'B';
        input[i - 1] = 'G';
        ++i;
      }

  cout << input << endl;

  return 0;
}