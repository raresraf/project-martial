#include <iostream>

using namespace std;

int main() {
  string s;
  std::cin >> s;
  int a = 0, b = 0, c = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      a++;
    } else if (s[i] == '2') {
      b++;
    } else if (s[i] == '3') {
      c++;
    }
  }
  for (int j = 0; j < a; j++) {
    s[2 * j] = '1';
  }
  for (int k = 0; k < b; k++) {
    s[2 * (a + k)] = '2';
  }
  for (int l = 0; l < c; l++) {
    s[2 * (a + b + l)] = '3';
  }
  std::cout << s;
  return 0;
}