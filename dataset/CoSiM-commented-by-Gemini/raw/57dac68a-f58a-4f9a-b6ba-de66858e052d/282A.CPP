#include <cstring>
#include <iostream>

using namespace std;
int main() {
  int t, x = 0;
  char str[4];
  cin >> t;
  for (; t > 0; t--) {
    cin >> str;
    if ((strcmp(str, "X++") == 0) || (strcmp(str, "++X") == 0))
      x = x + 1;
    if ((strcmp(str, "X--") == 0) || (strcmp(str, "--X") == 0))
      x = x - 1;
  }
  cout << x << endl;
  return 0;
}
