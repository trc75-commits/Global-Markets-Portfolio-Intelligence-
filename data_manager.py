
import sqlite3, json, os
SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS portfolios (
    name TEXT PRIMARY KEY,
    description TEXT,
    risk_profile TEXT,
    investment_horizon INTEGER,
    goal_notes TEXT,
    allocations TEXT,
    aum REAL,
    last_advice TEXT
);
'''
def init_db(path):
    conn=sqlite3.connect(path); cur=conn.cursor()
    cur.execute(SCHEMA_SQL); conn.commit(); conn.close()

def get_portfolios(path):
    if not os.path.exists(path): return []
    conn=sqlite3.connect(path); cur=conn.cursor()
    cur.execute("SELECT name,description,risk_profile,investment_horizon,goal_notes,allocations,aum,last_advice FROM portfolios")
    rows=cur.fetchall(); conn.close()
    out=[]
    for r in rows:
        name,desc,rp,ih,gn,alloc_json,aum,adv=r
        try: alloc=json.loads(alloc_json)
        except: alloc={}
        out.append({'name':name,'description':desc,'risk_profile':rp,
                    'investment_horizon':ih,'goal_notes':gn,'allocations':alloc,
                    'aum':aum,'last_advice':adv})
    return out

def load_portfolio(path,name):
    conn=sqlite3.connect(path); cur=conn.cursor()
    cur.execute("SELECT description,risk_profile,investment_horizon,goal_notes,allocations,aum,last_advice FROM portfolios WHERE name=?",(name,))
    r=cur.fetchone(); conn.close()
    if not r: return None
    desc,rp,ih,gn,alloc_json,aum,adv=r
    try: alloc=json.loads(alloc_json)
    except: alloc={}
    return {'description':desc,'risk_profile':rp,'investment_horizon':ih,
            'goal_notes':gn,'allocations':alloc,'aum':aum,'last_advice':adv}

def save_portfolio(path,name,data):
    conn=sqlite3.connect(path); cur=conn.cursor()
    cur.execute("REPLACE INTO portfolios VALUES (?,?,?,?,?,?,?,?)",
                (name,data.get('description',''),data.get('risk_profile','Balanced'),
                 data.get('investment_horizon',5),data.get('goal_notes',''),
                 json.dumps(data.get('allocations',{})),float(data.get('aum',100000)),
                 data.get('last_advice','')))
    conn.commit(); conn.close()

def delete_portfolio(path,name):
    conn=sqlite3.connect(path); cur=conn.cursor()
    cur.execute("DELETE FROM portfolios WHERE name=?",(name,))
    conn.commit(); conn.close()
