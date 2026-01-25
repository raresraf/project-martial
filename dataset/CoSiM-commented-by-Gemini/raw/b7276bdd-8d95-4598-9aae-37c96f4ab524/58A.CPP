#include <iostream>
#include <string>

using namespace std;

int main() {
  string str, str1 = "hello";
  cin >> str;

  int j = 0, pass = 0;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == str1[j]) {
      j++;
      pass++;

      if (pass == 5)
        break;
    }
  }

  if (pass == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
