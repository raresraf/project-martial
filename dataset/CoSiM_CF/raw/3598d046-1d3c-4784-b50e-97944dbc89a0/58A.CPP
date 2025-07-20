#include <iostream>
#include <stack>

using namespace std;
int main() {

  string str;
  string ref = "hello";
  int pass = 0;
  cin >> str;
  int j = 0;
  for (int i = 0; i < str.length(); i++) {

    if (str[i] == ref[j]) {
      j++;
      pass++;
    }
  }
  if (pass == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
