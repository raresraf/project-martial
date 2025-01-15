#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

bool periksa(vector<char> &dis, char a) {
  for (int j = 0; j < dis.size(); j++) {
    if (dis[j] == a) {
      return false;
    }
  }
  return true;
}

int main() {
  string kata;
  cin >> kata;

  vector<char> dis;
  for (int i = 0; i < kata.length(); i++) {
    if (dis.size() == 0) {
      dis.push_back(kata[i]);
    } else {
      if (periksa(dis, kata[i])) {
        dis.push_back(kata[i]);
      }
    }
  }
  if (dis.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}
