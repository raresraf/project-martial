#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  char words[n][100], lengths[n];
  for (int i = 0; i < n; i++) {
    cin >> words[i];
    lengths[i] = strlen(words[i]);
  }
  for (int j = 0; j < n; j++) {
    if (lengths[j] <= 10) {
      cout << words[j] << "\n";
    } else {
      cout << words[j][0] << (lengths[j] - 2) << words[j][lengths[j] - 1]
           << "\n";
    }
  }
  return 0;
}
