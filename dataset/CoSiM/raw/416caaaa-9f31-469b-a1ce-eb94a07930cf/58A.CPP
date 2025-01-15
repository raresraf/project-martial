#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string word = "hello";
  string type;
  cin >> type;
  int length = type.length();
  int a = 0;
  int i, count = 0;
  for (i = 0; i < length; i++) {
    if (type[i] == word[a]) {
      a++;
      count++;
    }
  }
  if (count == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
