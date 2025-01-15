#include <bits/stdc++.h>
using namespace std;

int main() {
  string input;
  int counter = 0;
  cin >> input;
  char tampung[input.size()];

  if (input.size() == 1) {
    cout << input;
  } else {
    for (int i = 0; i < input.size(); i++) {
      if (i % 2 == 0) {
        tampung[counter] = input[i];
        counter++;
      }
    }

    sort(tampung, tampung + strlen(tampung));
    counter = 0;
    for (int i = 0; i < input.size(); i++) {
      if (i % 2 == 0) {
        cout << tampung[counter];
        counter++;
      } else {
        cout << "+";
      }
    }
  }

  return 0;
}
