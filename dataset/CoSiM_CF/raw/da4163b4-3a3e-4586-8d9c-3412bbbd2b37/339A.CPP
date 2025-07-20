#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {

  string str, ans = "";
  cin >> str;
  int count1, count2, count3;
  count1 = count2 = count3 = 0;
  for (int i = 0; i < str.length(); i++) {

    if (str[i] != '+') {
      if (str[i] == '1')
        count1++;
      else if (str[i] == '2')
        count2++;
      else
        count3++;
    }
  }
  for (int x = 0; x < count1; x++)
    ans += "1+";
  for (int y = 0; y < count2; y++)
    ans += "2+";
  for (int z = 0; z < count3; z++)
    ans += "3+";

  ans.pop_back();
  cout << ans << endl;
  return 0;
}
