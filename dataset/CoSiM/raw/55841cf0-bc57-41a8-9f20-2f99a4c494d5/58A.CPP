#include <bits/stdc++.h>
using namespace std;

int main() {
  string input;
  int hello = 0;
  cin >> input;
  for (int i = 0; i < input.length(); i++) {
    if (hello == 5) {
      break;
    } else if (input[i] == 'o' && hello == 4) {
      hello++;
    } else if (input[i] == 'l' && hello == 3) {
      hello++;
    } else if (input[i] == 'l' && hello == 2) {
      hello++;
    } else if (input[i] == 'e' && hello == 1) {
      hello++;
    } else if (input[i] == 'h' && hello == 0) {
      hello++;
    }
  }
  if (hello == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
