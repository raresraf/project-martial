#include <iostream>
#include <set>

using namespace std;

int main() {
  string userName;
  set<char> distinctCharacters;

  getline(cin, userName);
  for (int i = 0; i < userName.size(); i++) {
    distinctCharacters.insert(userName[i]);
  }

  if (distinctCharacters.size() % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";

  return 0;
}
