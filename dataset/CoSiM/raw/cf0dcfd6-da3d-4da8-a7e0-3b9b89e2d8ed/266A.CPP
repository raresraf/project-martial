#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n;
  cin >> n;
  string s;
  cin >> s;

  int count = 0;
  for (int i = 0; i < n; i++) {
    if (i == n - 1)
      break;
    if (s[i] == s[i + 1])
      count++;
  }
  cout << count;
  return 0;
}
