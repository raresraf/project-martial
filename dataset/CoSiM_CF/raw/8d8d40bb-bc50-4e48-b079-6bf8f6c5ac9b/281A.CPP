
#include <iostream>
using namespace std;
int main() {

  char str[1001];
  cin >> str;
  cout << (char)toupper(str[0]);
  for (int i = 1; str[i] != '\0'; i++) {
    cout << (str[i]);
  }

  return 0;
}
