#include <cstring>
#include <iostream>

using namespace std;

int main() {
  char first[101];
  char second[101];

  cin >> first;
  cin >> second;

  for (int i = 0; i < strlen(first); i++) {
    if (first[i] == second[i] || first[i] == second[i] + 32 ||
        first[i] == second[i] - 32) {
      continue;
    } else {
      if (first[i] <= 'Z') {
        first[i] += 32;
      }
      if (second[i] <= 'Z') {
        second[i] += 32;
      }

      if (first[i] < second[i]) {
        cout << -1 << endl;
        return 0;
      }

      else {
        cout << 1 << endl;
        return 0;
      }
    }
  }
  cout << 0 << endl;
  return 0;
}