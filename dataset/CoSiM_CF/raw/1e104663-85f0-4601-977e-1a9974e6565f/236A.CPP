#include <iostream>
using namespace std;
main() {
  string str;
  cin >> str;
  int arr[26] = {0};
  int n = str.length();
  for (int i = 0; i < n; i++)
    arr[str[i] - 97] = 1;

  int x = 0;
  for (int i = 0; i < 26; i++) {
    if (arr[i] != 0)
      x = x + 1;
  }
  // cout<<"x "<<x<<endl;
  if (x % 2 != 0)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;
}
