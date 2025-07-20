/*
 *Link: https://codeforces.com/problemset/problem/110/A
 */

#include <iostream>
using namespace std;

int main() {
  string number;
  int count = 0;
  cin >> number;
  for (int i = 0; i < number.length(); i++) {
    if (number[i] == '4' || number[i] == '7')
      count++;
  }
  cout << (count == 4 || count == 7 ? "YES" : "NO") << endl;
}