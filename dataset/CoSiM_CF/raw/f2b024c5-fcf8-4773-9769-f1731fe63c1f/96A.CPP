#include <iostream>

using namespace std;

int main() {
  string players = "";
  string chance = "";
  string result = "";
  string s1 = "1111111";
  string s2 = "0000000";
  cin >> players;
  for (int i = 0; i < players.length(); i++) {
    chance += players[i];
  }
  if (chance.find(s2) != string::npos) {
    result = "YES";
  } else if (chance.find(s1) != string::npos)

  {
    result = "YES";
  } else {
    result = "NO";
  }

  cout << result << endl;
  return 0;
}
