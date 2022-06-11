#include <bits/stdc++.h>
using namespace std;
#define ll long long int

vector<int> g[1001];
vector<pair<ll,vector<ll>>> pt;

void dfs(ll st,ll e,ll vis[],vector<ll> rs,ll w) {
    rs.push_back(st);
    if(st == e) {
        pt.push_back({w*(rs.size()-1),rs});
        return;
    }
    for(auto u : g[st]) {
        if(vis[u] == 0) {
            vis[st] = 1;
            dfs(u,e,vis,rs,w);
            vis[st] = 0;
        }
    }
}

int main()
{
    ll n,m,t,c,u,v;
    cin>>n>>m>>t>>c;
    while(m--) {
        cin>>u>>v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    if(n == 1)
        cout<<0<<endl;
    else if(n == 2)
        cout<<t<<endl;
    else {
        vector<ll> rs;
        ll w = c;
        ll vis[n+1] = {0};
        dfs(1,n,vis,rs,w);
        if(pt.size() == 0)
            cout<<-1<<endl;
        else {
            sort(pt.begin(),pt.end());
            ll te = 0;
            ll nt = 0;
            for(int i=1; i<pt[0].second.size(); i++) {
                te += c + (nt-te);
                while(nt < te)
                    nt += t;
            }
            cout<<te<<endl;
        }//else
    }
    return 0;
}
