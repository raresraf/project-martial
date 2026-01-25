#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  char a[] = {'h', 'e', 'l', 'l', 'o'};
  int j = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == a[j]) {
      j++;
    }
  }
  if (j < 5)
    cout << "NO";
  else {
    cout << "YES";
  }
  return 0;
}