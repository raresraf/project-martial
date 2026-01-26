#include <bits/stdc++.h>
using namespace std;

int main() {
  string situation;
  cin >> situation;
  if (situation.size() < 7) {
    cout << "NO";
  } else {
    int counter = 1;
    for (int i = 1; i < situation.size(); i++) {
      if (situation[i] == situation[i - 1]) {
        counter += 1;
      } else if (counter < 7) {
        counter = 1;
      }
    }
    if (counter >= 7) {
      cout << "YES";
    } else {
      cout << "NO";
    }
  }

  return 0;
}
