#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  string user_name;
  cin >> user_name;
  int length = user_name.length(), temp, dis_length = 0, j;
  sort(user_name.begin(), user_name.end());

  for (j = 0; j < length; j++) {
    if (user_name[j] != user_name[j + 1])
      dis_length++;
    else
      continue;
  }

  if (dis_length % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
}
