#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int arr[26] = {0};
  string s;
  cin >> s;
  for (int i = 0; s[i]; i++) {
    arr[s[i] - 97] = 1;
  }
  int sum = 0;
  for (int i = 0; i < 26; i++) {
    sum += arr[i];
  }
  if (sum % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}