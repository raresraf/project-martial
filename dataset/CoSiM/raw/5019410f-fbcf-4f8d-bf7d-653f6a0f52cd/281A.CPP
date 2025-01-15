#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  // your code goes here
  string ans;
  cin >> ans;
  ans[0] = toupper(ans[0]);
  cout << ans << endl;
  return 0;
}