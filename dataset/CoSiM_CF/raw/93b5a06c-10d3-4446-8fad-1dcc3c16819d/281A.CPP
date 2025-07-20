#include <bits/stdc++.h>

using namespace std;

int main() {
  string input;
  cin >> input;
  if (islower(input[0])) {
    input[0] = input[0] - 32;
  }
  cout << input;
  return 0;
}
