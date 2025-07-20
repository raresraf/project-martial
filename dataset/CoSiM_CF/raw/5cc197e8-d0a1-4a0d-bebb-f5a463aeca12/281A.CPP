#include <iostream>
using namespace std;

int main() {

  char s[1001];
  cin >> s;
  if (s[0] >= 97)
    s[0] -= 32;

  cout << s << endl;

  return 0;
}