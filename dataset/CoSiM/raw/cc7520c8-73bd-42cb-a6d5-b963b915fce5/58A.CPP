#include <iostream>
#include <string>
#define LOG(x) cout << x << "\n"

using namespace std;

bool parseHello(string);

int main() {
  string str;
  cin >> str;
  parseHello(str) ? LOG("YES") : LOG("NO");
  return 0;
}

bool parseHello(string s) {
  char hello[] = {'h', 'e', 'l', 'l', 'o'};
  int i = 0;
  for (char c : s) {
    if (i < 5 && c == hello[i]) {
      i++;
    }
    if (i == 5) {
      return true;
    }
  }
  return false;
}