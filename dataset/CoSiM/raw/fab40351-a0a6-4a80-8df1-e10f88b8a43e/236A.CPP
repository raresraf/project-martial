// 236A - Boy or Girl

#include <iostream>

using namespace std;

int main() {
  string name;
  int i, d = 0, letter[26] = {0};

  cin >> name;

  for (i = 0; i < name.size(); i++)
    letter[name[i] - 'a']++;

  for (i = 0; i < 26; i++)
    if (letter[i] > 0)
      d++;

  if (d % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";

  return 0;
}
