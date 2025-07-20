#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  if (str[0] == '0')
    cout << "NO" << endl;
  else
    cout << "YES" << endl;
  return 0;
}
