#include <bits/stdc++.h>
#include <string>
using namespace std;
int main() {
  int n;
  cin >> n;
  while (n > 0) {
    string str;
    cin >> str;
    if (str.length() > 10) {

      int size = str.length() - 2;
      string temp = to_string(size);
      char first = str.at(0);
      char last = str.at(str.length() - 1);
      temp = first + temp + last;
      cout << temp << endl;
    } else {
      cout << str << endl;
    }
    n--;
  }
}
