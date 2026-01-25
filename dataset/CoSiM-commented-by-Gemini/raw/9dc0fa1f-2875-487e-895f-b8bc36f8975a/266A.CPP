#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int n;
  cin >> n >> s;
  int len = s.length();
  int count = 0;
  for (int i = 0; i < len; i++) {
    if (s[i] == s[i + 1]) {
      count++;
    }
  }
  cout << count;
  return 0;
}