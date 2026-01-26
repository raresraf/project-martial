#include <bits/stdc++.h>
#include <iostream>

int main() {
  std::string username;
  std::cin >> username;
  int pass = 0;

  std::sort(username.begin(), username.end());

  for (int i = 1; i < username.length(); i++) {
    if (username[i] != username[i - 1]) {
      pass++;
    }
  }

  /*for(int i = 0; i < username.length(); i++){
      for(int j = 0; j < i; j++){
          if(username[i] == username[j]){
              n++;     //zwiekszanie wartosci tych samych slow
          }
      }

      if(n == 0){
          pass++;
      }

      n = 0;
  }*/

  if (pass % 2 == 0) {
    std::cout << "IGNORE HIM!";
  } else {
    std::cout << "CHAT WITH HER!";
  }

  return 0;
}