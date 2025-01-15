#include <cstring>
#include <iostream>

using namespace std;
int main() {
  int t;
  char str[1000];
  cin >> t;
  for (; t > 0; t--) {
    cin >> str;
    if (strlen(str) <= 10)
      cout << str << endl;
    else
      cout << str[0] << strlen(str) - 2 << str[strlen(str) - 1] << endl;
  }
  return 0;
}
