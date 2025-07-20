#include <iostream>
using namespace std;

int main() {
  char s[101];
  cin >> s;
  char myw[] = {'h', 'e', 'l', 'l', 'o'};
  int t = 0;
  for (int i = 0; s[i] != '\0' && t < 5; i++) {
    if (myw[t] == s[i])
      t++;
  }
  if (t == 5)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}