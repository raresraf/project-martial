/*
 *Link: https://codeforces.com/problemset/problem/266/A
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string stones;
  int n, answer = 0;
  cin >> n >> stones;
  for (int i = 0; i < stones.length() - 1; i++) {
    if (stones[i] == stones[i + 1])
      answer++;
  }
  cout << answer << endl;
}