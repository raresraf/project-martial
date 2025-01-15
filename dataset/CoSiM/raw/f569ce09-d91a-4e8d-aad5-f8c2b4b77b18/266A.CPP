#include <iostream>
using namespace std;
int main() {
  int T, i = 0, k = 0;
  cin >> T;
  char s[100];
  cin >> s;
  while (i <= T - 2) {
    if (s[i] == s[i + 1])
      k++;
    i++;
  }
  cout << k;
  return 0;
}
